/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of the Dimensions class.
 */

#include <cstddef>

#ifndef OPENKALMAN_DIMENSIONS_HPP
#define OPENKALMAN_DIMENSIONS_HPP

namespace OpenKalman
{
  /**
   * \brief A structure representing the dimension associated with of a given index.
   * \details The dimension may or may not be known at compile time. If unknown at compile time, the size is set
   * at the time of construction and cannot be modified thereafter.
   * \tparam size The dimension (or <code>dynamic_size</code>, if not known at compile time)
   */
  template<std::size_t size = dynamic_size>
  struct Dimensions;


  /**
   * \overload
   * \brief The dimension or size associated with a given index known at compile time.
   */
  template<std::size_t size>
  struct Dimensions
  {
    /// The size of the dimension.
    static constexpr std::size_t value = size;

    /// \returns The size of the dimension as known at compile time
    constexpr std::size_t operator()() const
    {
      return size;
    }

    /// \returns The size of the dimension as known at compile time
    constexpr operator std::size_t() const
    {
      return size;
    }
  };


  /**
   * \overload
   * \brief The dimension or size associated with a given dynamic index known only at runtime.
   */
  template<>
  struct Dimensions<dynamic_size>
  {
  private:

    const std::size_t runtime_size;

  public:

    /// The size of the dimension (which in this case is <code>dynamic_size</code>).
    static constexpr std::size_t value = dynamic_size;

    Dimensions() = delete;

    /// Constructor, taking a dimension known at runtime.
    Dimensions(const std::size_t size) : runtime_size(size)
    {}

    /// \returns The size of the dimension as known at runtime
    const std::size_t operator()() const
    {
      return runtime_size;
    }

    /// \returns The size of the dimension as known at runtime
    operator std::size_t() const
    {
      return runtime_size;
    }
  };


#ifdef __cpp_concepts
  template<std::convertible_to<const std::size_t&> T>
#else
  template<typename T, std::enable_if_t<std::is_convertible<T, const std::size_t&>::value, int> = 0>
#endif
  Dimensions(T&&) -> Dimensions<dynamic_size>;


  namespace detail
  {
    template<typename T>
    struct is_index_descriptor : std::false_type {};

    template<std::size_t size>
    struct is_index_descriptor<Dimensions<size>> : std::true_type {};

    template<>
    struct is_index_descriptor<std::size_t> : std::true_type {};
  }


  /**
   * \brief Specifies that a type is a self-adjoint matrix.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept index_descriptor =
#else
  constexpr bool index_descriptor =
#endif
    detail::is_index_descriptor<std::decay_t<T>>::value;


  /**
   * \brief The dimension size of an \ref index_descriptor.
   */
#ifdef __cpp_concepts
  template<index_descriptor T>
#else
  template<typename T, typename = void>
#endif
  struct dimension_size_of;


#ifdef __cpp_concepts
  template<index_descriptor T> requires std::is_integral_v<std::decay_t<T>>
  struct dimension_size_of<T>
#else
  template<typename T>
  struct dimension_size_of<T, std::enable_if_t<index_descriptor<T> and std::is_integral_v<std::decay_t<T>>>>
#endif
    : std::integral_constant<std::size_t, dynamic_size> {};


#ifdef __cpp_concepts
  template<index_descriptor T> requires (not std::is_integral_v<std::decay_t<T>>)
  struct dimension_size_of<T>
#else
  template<typename T>
  struct dimension_size_of<T, std::enable_if_t<index_descriptor<T> and not std::is_integral_v<std::decay_t<T>>>>
#endif
    : std::integral_constant<std::size_t, std::decay_t<T>::value> {};


  /**
   * \brief Helper template for \ref dimension_size_of.
   */
#ifdef __cpp_concepts
  template<index_descriptor T>
#else
  template<typename T>
#endif
  constexpr auto dimension_size_of_v = dimension_size_of<std::decay_t<T>>::value;


#ifdef __cpp_concepts
  template<index_descriptor T>
#else
  template<typename T, std::enable_if_t<index_descriptor<T>, int> = 0>
#endif
  constexpr std::size_t
  get_dimension_size_of(const T& t)
  {
    return t;
  }

} // namespace OpenKalman


#endif //OPENKALMAN_DIMENSIONS_HPP
