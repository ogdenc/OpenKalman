/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition for \ref internal::tuple_wrapper.
 */

#ifndef OPENKALMAN_TUPLE_WRAPPER_HPP
#define OPENKALMAN_TUPLE_WRAPPER_HPP

#include <functional>
#include "basics/basics.hpp"
#include "collections/traits/size_of.hpp"
#include "collections/functions/get.hpp"
#include "collections/traits/collection_element.hpp"
#include "collections/concepts/viewable_tuple_like.hpp"
#include "collections/views/internal/movable_wrapper.hpp"

namespace OpenKalman::collections::internal
{
  /**
   * \internal
   * \brief A movable wrapper for any value, whether lvalue or rvalue.
   * \details The wrapper is guaranteed to be std::movable.
   * It is also guaranteed to be copyable if T is either copy_constructible or an lvalue reference.
   */
#ifdef __cpp_concepts
  template<viewable_tuple_like T>
#else
  template<typename T, typename = void>
#endif
  struct tuple_wrapper
  {
  private:

#ifndef __cpp_concepts
    static_assert(viewable_tuple_like<T>);
#endif

    using Seq = std::make_index_sequence<size_of_v<T>>;


    template<typename U, std::size_t...i>
    static constexpr decltype(auto)
    fill_tuple(U&& u, std::index_sequence<i...>)
    {
      return std::tuple { movable_wrapper {collections::get<i>(std::forward<U>(u))}...};
    }


    using T_ = OpenKalman::internal::movable_box<decltype(fill_tuple(std::declval<T>(), Seq{}))>;

  public:

    /**
     * \brief The wrapped type
     */
    using type = T;


    /**
     * \brief Default constructor.
     */
    constexpr
#ifdef __cpp_lib_concepts
    tuple_wrapper() noexcept(std::is_nothrow_default_constructible_v<T_>) requires std::default_initializable<T_> = default;
#else
    tuple_wrapper() noexcept(std::is_nothrow_default_constructible_v<T_>) = default;
#endif


    /**
     * \brief Construct from an rvalue reference.
     */
    explicit constexpr
    tuple_wrapper(T&& t) noexcept : t_ {fill_tuple(std::forward<T>(t), Seq{})} {}


    /**
     * \brief Get an element.
     */
#ifdef __cpp_explicit_this_parameter
    template<std::size_t i>
    constexpr decltype(auto)
    get(this auto&& self) noexcept
    {
      static_assert(i < size_of_v<T>, "Index out of range");
      return collections::get<i>(*(std::forward<decltype(self)>(self).t_).get());
    }
#else
    template<std::size_t i>
    constexpr decltype(auto) get() & noexcept
    {
      static_assert(i < size_of_v<T>, "Index out of range");
      return collections::get<i>(*t_).get();
    }

    /// \overload
    template<std::size_t i>
    constexpr decltype(auto) get() const & noexcept
    {
      static_assert(i < size_of_v<T>, "Index out of range");
      return collections::get<i>(*t_).get();
    }

    /// \overload
    template<std::size_t i>
    constexpr decltype(auto) get() && noexcept
    {
      static_assert(i < size_of_v<T>, "Index out of range");
      return collections::get<i>(*std::move(t_)).get();
    }

    /// \overload
    template<std::size_t i>
    constexpr decltype(auto) get() const && noexcept
    {
      static_assert(i < size_of_v<T>, "Index out of range");
      return collections::get<i>(*std::move(t_)).get();
    }
#endif

  private:

    T_ t_;

  };


  /**
   * \overload
   * \internal
   * \brief Specialization in which T is a move_constructible object.
   */
#ifdef __cpp_concepts
  template<viewable_tuple_like T> requires std::move_constructible<T> and std::is_object_v<T>
  struct tuple_wrapper<T>
#else
  template<typename T>
  struct tuple_wrapper<T, std::enable_if_t<viewable_tuple_like<T> and stdex::move_constructible<T> and std::is_object_v<T>>>
#endif
  {
  private:

    using T_ = OpenKalman::internal::movable_box<T>;

  public:

    /**
     * \brief The wrapped type
     */
    using type = T;


    /**
     * \brief Default constructor.
     */
    constexpr
#ifdef __cpp_lib_concepts
    tuple_wrapper() noexcept(std::is_nothrow_default_constructible_v<T_>) requires std::default_initializable<T_> = default;
#else
    tuple_wrapper() noexcept(std::is_nothrow_default_constructible_v<T_>) = default;
#endif


    /**
     * \brief Construct from an rvalue reference.
     */
    explicit constexpr
    tuple_wrapper(T&& t) : t_ {std::move(t)} {}


    /**
     * \brief Retrieve the stored value.
     */
    template<std::size_t i>
    constexpr decltype(auto) get() & noexcept
    {
      static_assert(i < size_of_v<T>, "Index out of range");
      return  collections::get<i>(t_.operator*());
    }

    /// \overload
    template<std::size_t i>
    constexpr decltype(auto) get() const & noexcept
    {
      static_assert(i < size_of_v<T>, "Index out of range");
      return  collections::get<i>(t_.operator*());
    }

    /// \overload
    template<std::size_t i>
    constexpr decltype(auto) get() && noexcept
    {
      static_assert(i < size_of_v<T>, "Index out of range");
      return  collections::get<i>(std::move(t_.operator*()));
    }

    /// \overload
    template<std::size_t i>
    constexpr decltype(auto) get() const && noexcept
    {
      static_assert(i < size_of_v<T>, "Index out of range");
      return  collections::get<i>(std::move(t_.operator*()));
    }

  private:

    T_ t_;

  };


  /**
   * \overload
   * \internal
   * \brief Specialization for reference types
   */
#ifdef __cpp_concepts
  template<viewable_tuple_like T>
  struct tuple_wrapper<T&>
#else
  template<typename T>
  struct tuple_wrapper<T&, std::enable_if_t<viewable_tuple_like<T>>>
#endif
  {
  private:

    using T_ = stdex::reference_wrapper<std::remove_reference_t<T>>;

  public:

    /**
     * \brief The wrapped type
     */
    using type = T&;


    /**
     * \brief Construct from an lvalue reference.
     */
    explicit constexpr
    tuple_wrapper(T& t) : t_ {t} {}


    /**
     * \brief Get an element.
     */
    template<std::size_t i>
    constexpr decltype(auto) get() const noexcept
    {
      static_assert(i < size_of_v<T>, "Index out of range");
      return collections::get<i>(t_.get());
    }

  private:

    T_ t_;

  };


  /**
   * \internal
   * \brief deduction guide
   */
  template<typename T>
  tuple_wrapper(T&&) -> tuple_wrapper<T>;

}


namespace std
{
  template<typename Tup>
  struct tuple_size<OpenKalman::collections::internal::tuple_wrapper<Tup>>
    : OpenKalman::collections::size_of<std::decay_t<Tup>> {};

  template<std::size_t i, typename Tup>
  struct tuple_element<i, OpenKalman::collections::internal::tuple_wrapper<Tup>>
    : OpenKalman::collections::collection_element<i, std::decay_t<Tup>> {};
}

#endif
