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
#include "collections/collections.hpp"
#include "coordinates/interfaces/coordinate_descriptor_traits.hpp"
#include "coordinates/functions/to_stat_space.hpp"
#include "coordinates/functions/from_stat_space.hpp"
#include "coordinates/functions/wrap.hpp"
#include "coordinates/functions/internal/get_descriptor_hash_code.hpp"
#include "coordinates/functions/internal/get_descriptor_dimension.hpp"
#include "coordinates/functions/internal/get_descriptor_stat_dimension.hpp"

namespace OpenKalman::coordinates
{
  /**
   * \internal
   * \brief A type representing any \ref coordinates::descriptor object.
   * \tparam S The value type for elements associated with this \ref coordinates::pattern object.
   */
#ifdef __cpp_concepts
  template<values::number S = double> requires std::same_as<S, std::decay_t<S>>
#else
  template<typename S = double>
#endif
  struct Any;


  namespace detail
  {
    /**
     * \brief Tests whether the argument is an instance of of type \ref Any.
     */
    template<typename T>
    struct is_Any : std::false_type {};

    template<typename S>
    struct is_Any<Any<S>> : std::true_type {};
  }


#ifdef __cpp_concepts
  template<values::number S> requires std::same_as<S, std::decay_t<S>>
#else
  template<typename S>
#endif
  struct Any
  {
  private:

    using Getter = std::function<S(std::size_t)>;

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
      template<typename Arg>
      explicit Derived(Arg&& arg) : my_t(std::forward<Arg>(arg)) {}

      [[nodiscard]] std::size_t dimension() const final { return internal::get_descriptor_dimension(my_t); }

      [[nodiscard]] std::size_t stat_dimension() const final { return internal::get_descriptor_stat_dimension(my_t); }

      [[nodiscard]] bool is_euclidean() const final { return internal::get_descriptor_is_euclidean(my_t); }

      [[nodiscard]] std::size_t hash_code() const final { return internal::get_descriptor_hash_code(my_t); }

      [[nodiscard]] Getter to_stat_space(Getter g) const final
      {
        if constexpr (euclidean_pattern<T>)
        {
          return std::move(g);
        }
        else
        {
          return [stat_data = coordinates::to_stat_space(my_t, collections::views::generate(std::move(g), get_dimension(my_t)))]
            (std::size_t i) -> S { return collections::get_element(stat_data, i); };
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
          return [data = coordinates::from_stat_space(my_t, collections::views::generate(std::move(g), get_stat_dimension(my_t)))]
            (std::size_t i) -> S { return collections::get_element(data, i); };
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
          return [data = coordinates::wrap(my_t, collections::views::generate(std::move(g), get_dimension(my_t)))]
            (std::size_t i) -> S { return collections::get_element(data, i); };
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
    template <descriptor Arg> requires (not detail::is_Any<std::decay_t<Arg>>::value)
#else
    template<typename Arg, std::enable_if_t<descriptor<Arg> and (not detail::is_Any<std::decay_t<Arg>>::value), int> = 0>
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
  template<typename S>
  struct coordinate_descriptor_traits<coordinates::Any<S>>
  {
  private:

    using T = coordinates::Any<S>;

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
        t.mBase->to_stat_space([d](std::size_t i){ return collections::get_element(std::get<0>(d), i); }),
        t.mBase->stat_dimension());
    };


    static constexpr auto
    from_stat_space = [](const T& t, auto&& data_view)
    {
      auto d = std::make_tuple(std::forward<decltype(data_view)>(data_view));
      return collections::views::generate(
        t.mBase->from_stat_space([d](std::size_t i){ return collections::get_element(std::get<0>(d), i); }),
        t.mBase->dimension());
    };


    static constexpr auto
    wrap = [](const T& t, auto&& data_view)
    {
      auto d = std::make_tuple(std::forward<decltype(data_view)>(data_view));
      return collections::views::generate(
        t.mBase->wrap([d](std::size_t i){ return collections::get_element(std::get<0>(d), i); }),
        t.mBase->dimension());
    };

  };

}


namespace std
{
  template<typename Scalar1, typename Scalar2>
  struct common_type<OpenKalman::coordinates::Any<Scalar1>, OpenKalman::coordinates::Any<Scalar2>>
    : std::conditional_t<
        OpenKalman::stdex::common_with<Scalar1, Scalar2>,
        OpenKalman::stdex::type_identity<OpenKalman::coordinates::Any<
          typename std::conditional_t<
            OpenKalman::stdex::common_with<Scalar1, Scalar2>,
            common_type<Scalar1, Scalar2>,
            OpenKalman::stdex::type_identity<double>>::type>>,
        std::monostate> {};


  template<typename S, typename T>
  struct common_type<OpenKalman::coordinates::Any<S>, T>
    : std::conditional_t<
      OpenKalman::coordinates::descriptor<T>,
      OpenKalman::stdex::type_identity<OpenKalman::coordinates::Any<S>>,
      std::monostate> {};
}

#endif
