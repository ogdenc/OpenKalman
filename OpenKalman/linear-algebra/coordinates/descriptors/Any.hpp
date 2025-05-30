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
 * \brief Definition of the coordinates::Any class.
 */

#ifndef OPENKALMAN_ANYATOMICVECTORTYPES_HPP
#define OPENKALMAN_ANYATOMICVECTORTYPES_HPP

#include <memory>
#include "values/concepts/number.hpp"
#include "linear-algebra/coordinates/interfaces/coordinate_descriptor_traits.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"
#include "linear-algebra/coordinates/functions/to_stat_space.hpp"
#include "linear-algebra/coordinates/functions/from_stat_space.hpp"
#include "linear-algebra/coordinates/functions/get_wrapped_component.hpp"
#include "linear-algebra/coordinates/functions/set_wrapped_component.hpp"

namespace OpenKalman::coordinates
{
  /**
   * \internal
   * \brief A type representing any \ref coordinates::descriptor object.
   * \tparam Scalar The scalar type for elements associated with this \ref coordinates::pattern object.
   */
#ifdef __cpp_concepts
  template<values::number Scalar = double>
#else
  template<typename Scalar = double>
#endif
  struct Any
  {
  private:

    using Getter = std::function<Scalar(std::size_t)>;
    using Setter = std::function<void(const Scalar&, std::size_t)>;

    struct Base
    {
      virtual ~Base() = default;
      [[nodiscard]] virtual std::size_t dimension() const = 0;
      [[nodiscard]] virtual std::size_t stat_dimension() const = 0;
      [[nodiscard]] virtual bool is_euclidean() const = 0;
      [[nodiscard]] virtual std::size_t hash_code() const = 0;
      [[nodiscard]] virtual Scalar to_stat_space(const Getter& g, std::size_t euclidean_local_index) const = 0;
      [[nodiscard]] virtual Scalar from_stat_space(const Getter& g, std::size_t local_index) const = 0;
      [[nodiscard]] virtual Scalar get_wrapped_component(const Getter& g, std::size_t local_index) const = 0;
      virtual void set_wrapped_component(const Setter& s, const Getter& g, const Scalar& x, std::size_t local_index) const = 0;
    };


    template <typename T>
    struct Derived : Base
    {
      template<typename Arg>
      explicit Derived(Arg&& arg) : my_t(std::forward<Arg>(arg)) {}

      [[nodiscard]] std::size_t dimension() const final { return get_dimension(my_t); }

      [[nodiscard]] std::size_t stat_dimension() const final { return get_stat_dimension(my_t); }

      [[nodiscard]] bool is_euclidean() const final { return get_is_euclidean(my_t); }

      [[nodiscard]] std::size_t hash_code() const final { return internal::get_hash_code(my_t); }

      [[nodiscard]] Scalar to_stat_space(const Getter& g, std::size_t euclidean_local_index) const final
      {
        return coordinates::to_stat_space(my_t, g, euclidean_local_index);
      }

      [[nodiscard]] Scalar from_stat_space(const Getter& g, std::size_t local_index) const final
      {
        return coordinates::from_stat_space(my_t, g, local_index);
      }

      [[nodiscard]] Scalar get_wrapped_component(const Getter& g, std::size_t local_index) const final
      {
        return coordinates::get_wrapped_component(my_t, g, local_index);
      }

      void set_wrapped_component(const Setter& s, const Getter& g, const Scalar& x, std::size_t local_index) const final
      {
        coordinates::set_wrapped_component(my_t, s, g, x, local_index);
      }

    private:

      T my_t;
    };

  public:

    /**
     * \brief Construct from a \ref coordinates::descriptor.
     */
#ifdef __cpp_concepts
    template <descriptor Arg>
#else
    template<typename Arg, std::enable_if_t<descriptor<Arg>, int> = 0>
#endif
    explicit constexpr
    Any(Arg&& arg) : mBase {std::make_shared<Derived<std::decay_t<Arg>>>(std::forward<Arg>(arg))} {}

  private:

    const std::shared_ptr<Base> mBase;

#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename>
#endif
    friend struct interface::coordinate_descriptor_traits;

  };


} // namespace OpenKalman::coordinates


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
    using Getter = std::function<Scalar(std::size_t)>;
    using Setter = std::function<void(const Scalar&, std::size_t)>;

  public:

    static constexpr bool is_specialized = true;


    using scalar_type = Scalar;


    static constexpr auto
    dimension(const T& t) { return t.mBase->dimension(); }


    static constexpr auto
    stat_dimension(const T& t) { return t.mBase->stat_dimension(); }


    static constexpr auto
    is_euclidean(const T& t) { return t.mBase->is_euclidean(); }


    static constexpr std::size_t
    hash_code(const T& t) { return t.mBase->hash_code(); }


#ifdef __cpp_concepts
    static constexpr values::value auto
    to_euclidean_component(const T& t, const auto& g, const values::index auto& euclidean_local_index)
    requires requires(std::size_t i){ {g(i)} -> std::convertible_to<scalar_type>; }
#else
    template<typename Getter, typename L, std::enable_if_t<values::index<L> and
      std::is_convertible_v<typename std::invoke_result<const Getter&, std::size_t>::type, scalar_type> and values::index<L>, int> = 0>
    static constexpr auto
    to_euclidean_component(const T& t, const Getter& g, const L& euclidean_local_index)
#endif
    {
      return t.mBase->to_stat_space(g, euclidean_local_index);
    }


#ifdef __cpp_concepts
    static constexpr values::value auto
    from_euclidean_component(const T& t, const auto& g, const values::index auto& local_index)
    requires requires(std::size_t i){ {g(i)} -> std::convertible_to<scalar_type>; }
#else
    template<typename Getter, typename L, std::enable_if_t<values::index<L> and
      std::is_convertible_v<typename std::invoke_result<const Getter&, std::size_t>::type, scalar_type>, int> = 0>
    static constexpr auto
    from_euclidean_component(const T& t, const Getter& g, const L& local_index)
#endif
    {
      return t.mBase->from_stat_space(g, local_index);
    }


#ifdef __cpp_concepts
    static constexpr values::value auto
    get_wrapped_component(const T& t, const auto& g, const values::index auto& local_index)
    requires requires(std::size_t i){ {g(i)} -> std::convertible_to<scalar_type>; }
#else
    template<typename Getter, typename L, std::enable_if_t<values::index<L> and
      std::is_convertible_v<typename std::invoke_result<const Getter&, std::size_t>::type, scalar_type>, int> = 0>
    static constexpr auto
    get_wrapped_component(const T& t, const Getter& g, const L& local_index)
#endif
    {
      return t.mBase->get_wrapped_component(g, local_index);
    }


#ifdef __cpp_concepts
    static constexpr void
    set_wrapped_component(const T& t, const auto& s, const auto& g, const scalar_type& x, const values::index auto& local_index)
    requires requires(std::size_t i){ s(x, i); s(g(i), i); }
#else
    template<typename Setter, typename Getter, typename L, std::enable_if_t<values::index<L> and
      std::is_invocable<const Setter&, const scalar_type&, std::size_t>::value and
      std::is_invocable<const Setter&, typename std::invoke_result<const Getter&, std::size_t>::type, std::size_t>::value, int> = 0>
    static constexpr void
    set_wrapped_component(const T& t, const Setter& s, const Getter& g, const scalar_type& x, const L& local_index)
#endif
    {
      t.mBase->set_wrapped_component(s, g, x, local_index);
    }

  };

} // namespace OpenKalman::interface


namespace std
{
#ifdef __cpp_concepts
  template<typename Scalar1, std::common_with<Scalar1> Scalar2>
#else
  template<typename Scalar1, typename Scalar2>
#endif
  struct common_type<OpenKalman::coordinates::Any<Scalar1>, OpenKalman::coordinates::Any<Scalar2>>
  {
    using type = OpenKalman::coordinates::Any<common_type_t<Scalar1, Scalar2>>;
  };


#ifdef __cpp_concepts
  template<typename Scalar, OpenKalman::coordinates::descriptor U>
#else
  template<typename Scalar, typename U>
#endif
  struct common_type<OpenKalman::coordinates::Any<Scalar>, U>
  {
    using type = OpenKalman::coordinates::Any<Scalar>;
  };


#ifdef __cpp_concepts
  template<OpenKalman::coordinates::descriptor T, typename Scalar>
#else
  template<typename T, typename Scalar>
#endif
  struct common_type<T, OpenKalman::coordinates::Any<Scalar>>
  {
    using type = OpenKalman::coordinates::Any<Scalar>;
  };
}

#endif //OPENKALMAN_ANYATOMICVECTORTYPES_HPP
