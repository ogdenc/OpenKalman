/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definition of \ref coordinates::Any.
 */

#ifndef OPENKALMAN_DESCRIPTOR_ANY_HPP
#define OPENKALMAN_DESCRIPTOR_ANY_HPP

#include <memory>
#include "values/concepts/number.hpp"
#include "collections/views/generate.hpp"
#include "linear-algebra/coordinates/interfaces/coordinate_descriptor_traits.hpp"
#include "linear-algebra/coordinates/functions/to_stat_space.hpp"
#include "linear-algebra/coordinates/functions/from_stat_space.hpp"
#include "linear-algebra/coordinates/functions/wrap.hpp"
#include "linear-algebra/coordinates/functions/internal/get_descriptor_hash_code.hpp"
#include "linear-algebra/coordinates/functions/internal/get_descriptor_dimension.hpp"
#include "linear-algebra/coordinates/functions/internal/get_descriptor_stat_dimension.hpp"
#include "linear-algebra/coordinates/functions/comparison-operators.hpp"

namespace OpenKalman::coordinates
{
  /**
   * \internal
   * \brief A type representing any \ref coordinates::descriptor object.
   * \tparam Scalar The scalar type for elements associated with this \ref coordinates::pattern object.
   */
#ifdef __cpp_concepts
  template<values::number Scalar = double> requires std::same_as<Scalar, std::decay_t<Scalar>>
#else
  template<typename Scalar = double>
#endif
  struct Any;


  namespace internal
  {
    /**
     * \brief Tests whether the argument is an instance of of type \ref Any.
     */
    template<typename T>
    struct is_Any : std::false_type { using scalar_type = double; };

    template<typename Scalar>
    struct is_Any<Any<Scalar>> : std::true_type { using scalar_type = Scalar; };
  }


#ifdef __cpp_concepts
  template<values::number Scalar> requires std::same_as<Scalar, std::decay_t<Scalar>>
#else
  template<typename Scalar>
#endif
  struct Any
  {
  private:

    using Getter = std::function<Scalar(std::size_t)>;

    struct Base
    {
      virtual ~Base() = default;
      [[nodiscard]] virtual std::size_t dimension() const = 0;
      [[nodiscard]] virtual std::size_t stat_dimension() const = 0;
      [[nodiscard]] virtual bool is_euclidean() const = 0;
      [[nodiscard]] virtual std::size_t hash_code() const = 0;
      [[nodiscard]] virtual Getter to_stat_space(Getter g) const = 0;
      [[nodiscard]] virtual Getter from_stat_space(Getter g) const = 0;
      [[nodiscard]] virtual Getter wrap(Getter g) const = 0;
    };


    template <typename T>
    struct Derived : Base
    {
      static_assert(descriptor<T> and not coordinates::internal::is_Any<T>::value);

      template<typename Arg>
      explicit Derived(Arg&& arg) : my_t(std::forward<Arg>(arg)) {}

      [[nodiscard]] std::size_t dimension() const final { return coordinates::internal::get_descriptor_dimension(my_t); }

      [[nodiscard]] std::size_t stat_dimension() const final { return coordinates::internal::get_descriptor_stat_dimension(my_t); }

      [[nodiscard]] bool is_euclidean() const final { return coordinates::internal::get_descriptor_is_euclidean(my_t); }

      [[nodiscard]] std::size_t hash_code() const final { return coordinates::internal::get_descriptor_hash_code(my_t); }

      [[nodiscard]] Getter to_stat_space(Getter g) const final
      {
        if constexpr (euclidean_pattern<T>)
        {
          return std::move(g);
        }
        else
        {
          auto stat_data = coordinates::to_stat_space(my_t, collections::views::generate(std::move(g), get_dimension(my_t)));
          return [stat_data](std::size_t i) -> Scalar { return collections::get(stat_data, i); };
        }
      }

      [[nodiscard]] Getter from_stat_space(Getter g) const final
      {
        if constexpr (euclidean_pattern<T>)
        {
          return std::move(g);
        }
        else
        {
          auto data = coordinates::from_stat_space(my_t, collections::views::generate(std::move(g), get_stat_dimension(my_t)));
          return [data](std::size_t i) -> Scalar { return collections::get(data, i); };
        }
      }

      [[nodiscard]] Getter wrap(Getter g) const final
      {
        if constexpr (euclidean_pattern<T>)
        {
          return std::move(g);
        }
        else
        {
          auto data = coordinates::wrap(my_t, collections::views::generate(std::move(g), get_dimension(my_t)));
          return [data](std::size_t i) -> Scalar { return collections::get(data, i); };
        }
      }

    private:

      T my_t;
    };

  public:

    /**
     * \brief Construct from a \ref coordinates::descriptor.
     */
#ifdef __cpp_concepts
    template <descriptor Arg> requires (not coordinates::internal::is_Any<std::decay_t<Arg>>::value)
#else
    template<typename Arg, std::enable_if_t<descriptor<Arg> and (not coordinates::internal::is_Any<std::decay_t<Arg>>::value), int> = 0>
#endif
    constexpr
    Any(Arg&& arg) : mBase {std::make_shared<Derived<std::decay_t<Arg>>>(std::forward<Arg>(arg))} {}


#ifndef __cpp_concepts
    // Addresses an issue with a version of clang in c++17
    constexpr Any() : mBase {std::make_shared<Derived<std::integral_constant<std::size_t, 0>>>(std::integral_constant<std::size_t, 0>{})} {}
#endif

  private:

    const std::shared_ptr<Base> mBase;

#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename>
#endif
    friend struct interface::coordinate_descriptor_traits;

  };

}


namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief traits for coordinates::Any.
   */
  template<typename Scalar>
  struct coordinate_descriptor_traits<coordinates::Any<Scalar>>
  {
  private:

    using T = coordinates::Any<Scalar>;

  public:

    static constexpr bool is_specialized = true;


    static constexpr auto dimension = [](const T& t) -> std::size_t { return t.mBase->dimension(); };


    static constexpr auto stat_dimension = [](const T& t) -> std::size_t { return t.mBase->stat_dimension(); };


    static constexpr auto is_euclidean = [](const T& t) -> bool { return t.mBase->is_euclidean(); };


    static constexpr auto hash_code = [](const T& t) -> std::size_t { return t.mBase->hash_code(); };


    static constexpr auto
    to_stat_space = [](const T& t, auto&& data_view)
    {
      auto d = std::make_tuple(std::forward<decltype(data_view)>(data_view));
      return collections::views::generate(
        t.mBase->to_stat_space([d](std::size_t i){ return collections::get(std::get<0>(d), i); }),
        t.mBase->stat_dimension());
    };


    static constexpr auto
    from_stat_space = [](const T& t, auto&& data_view)
    {
      auto d = std::make_tuple(std::forward<decltype(data_view)>(data_view));
      return collections::views::generate(
        t.mBase->from_stat_space([d](std::size_t i){ return collections::get(std::get<0>(d), i); }),
        t.mBase->dimension());
    };


    static constexpr auto
    wrap = [](const T& t, auto&& data_view)
    {
      auto d = std::make_tuple(std::forward<decltype(data_view)>(data_view));
      return collections::views::generate(
        t.mBase->wrap([d](std::size_t i){ return collections::get(std::get<0>(d), i); }),
        t.mBase->dimension());
    };

  };

}


namespace std
{
  template<typename Scalar1, typename Scalar2>
  struct common_type<OpenKalman::coordinates::Any<Scalar1>, OpenKalman::coordinates::Any<Scalar2>>
    : std::conditional_t<
        OpenKalman::stdcompat::common_with<Scalar1, Scalar2>,
        OpenKalman::stdcompat::type_identity<OpenKalman::coordinates::Any<
          typename std::conditional_t<
            OpenKalman::stdcompat::common_with<Scalar1, Scalar2>,
            common_type<Scalar1, Scalar2>,
            OpenKalman::stdcompat::type_identity<double>>::type>>,
        std::monostate> {};


  template<typename Scalar, typename T>
  struct common_type<OpenKalman::coordinates::Any<Scalar>, T>
    : std::conditional_t<
      OpenKalman::coordinates::descriptor<T>,
      OpenKalman::stdcompat::type_identity<OpenKalman::coordinates::Any<Scalar>>,
      std::monostate> {};
}

#endif
