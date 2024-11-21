/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for \ref make_constant.
 */

#ifndef OPENKALMAN_MAKE_CONSTANT_HPP
#define OPENKALMAN_MAKE_CONSTANT_HPP

namespace OpenKalman
{

  /**
   * \brief Make a constant object based on a particular library object.
   * \details A constant object is a matrix or tensor in which every component is the same scalar value.
   * \tparam T An \indexible object (matrix or tensor) from a particular library. Its shape and contents are irrelevant.
   * \tparam C A \ref value::scalar
   * \tparam Descriptors A \ref vector_space_descriptor_collection defining the dimensions of each index.
   */
#ifdef __cpp_concepts
  template<indexible T, value::scalar C, vector_space_descriptor_collection Descriptors>
  constexpr constant_matrix auto
#else
  template<typename T, typename C, typename Ds, std::enable_if_t<
    indexible<T> and value::scalar<C> and vector_space_descriptor_collection<Ds>, int> = 0>
  constexpr auto
#endif
  make_constant(C&& c, Descriptors&& descriptors)
  {
    decltype(auto) d = internal::remove_trailing_1D_descriptors(std::forward<Descriptors>(descriptors));
    using D = decltype(d);
    using Trait = interface::library_interface<std::decay_t<T>>;

    if constexpr (euclidean_vector_space_descriptor_collection<D> and interface::make_constant_defined_for<T, C&&, D>)
    {
      return Trait::template make_constant(std::forward<C>(c), std::forward<D>(d));
    }
    else if constexpr (interface::make_constant_defined_for<T, C&&, decltype(internal::to_euclidean_vector_space_descriptor_collection(d))>)
    {
      auto ed = internal::to_euclidean_vector_space_descriptor_collection(d);
      return make_vector_space_adapter(Trait::template make_constant(std::forward<C>(c), ed), std::forward<D>(d));
    }
    else
    {
      // Default behavior if interface function not defined:
      using Scalar = std::decay_t<decltype(value::to_number(c))>;
      using U = std::decay_t<decltype(make_dense_object<T, Layout::none, Scalar>(d))>;
      return ConstantAdapter<U, std::decay_t<C>> {std::forward<C>(c), std::forward<D>(d)};
    }
  }


  /**
   * \overload
   * \brief \ref vector_space_descriptors are specified as arguments.
   */
#ifdef __cpp_concepts
  template<indexible T, value::scalar C, vector_space_descriptor...Ds>
  constexpr constant_matrix auto
#else
  template<typename T, typename C, typename...Ds, std::enable_if_t<
    indexible<T> and value::scalar<C> and (vector_space_descriptor<Ds> and ...), int> = 0>
  constexpr auto
#endif
  make_constant(C&& c, Ds&&...ds)
  {
    return make_constant<T>(std::forward<C>(c), std::tuple {std::forward<Ds>(ds)...});
  }


  /**
   * \overload
   * \brief Make a new constant object based on a library object.
   * \tparam T The object on which the new matrix is patterned. This need not itself be constant, as only
   * its dimensions are used.
   * \tparam C A \ref value::scalar.
   */
#ifdef __cpp_concepts
  template<indexible T, value::scalar C>
  constexpr constant_matrix auto
#else
  template<typename T, typename C, std::enable_if_t<indexible<T> and value::scalar<C>, int> = 0>
  constexpr auto
#endif
  make_constant(const T& t, C&& c)
  {
    return make_constant<T>(std::forward<C>(c), all_vector_space_descriptors(t));
  }


  /**
   * \overload
   * \brief Make a compile-time constant based on a particular library object and a scalar constant value known at compile time
   * \tparam T A matrix or tensor from a particular library.
   * \tparam C A \ref value::scalar for the new zero matrix. Must be constructible from {constant...}
   * \tparam constant A constant or set of coefficients in a vector space defining a constant
   * (e.g., real and imaginary parts of a complex number).
   * \param Ds A \ref vector_space_descriptor_collection defining the dimensions of each index.
   */
#ifdef __cpp_concepts
  template<indexible T, value::scalar C, auto...constant, vector_space_descriptor_collection Ds> requires
    ((value::static_scalar<C> and sizeof...(constant) == 0) or requires { C {constant...}; })
  constexpr constant_matrix auto
#else
  template<typename T, typename C, auto...constant, typename Ds, std::enable_if_t<
    indexible<T> and value::scalar<C> and vector_space_descriptor_collection<Ds> and
    ((value::static_scalar<C> and sizeof...(constant) == 0) or
      std::is_constructible<C, decltype(constant)...>::value), int> = 0>
  constexpr auto
#endif
  make_constant(Ds&& ds)
  {
    using Scalar = std::decay_t<decltype(value::to_number(std::declval<C>()))>;
    if constexpr (sizeof...(constant) == 0)
      return make_constant<T>(C{}, std::forward<Ds>(ds));
    else
      return make_constant<T>(value::StaticScalar<Scalar, constant...>{}, std::forward<Ds>(ds));
  }


  /**
   * \overload
   * \brief Same as above, except that the scalar type is derived from the constant template parameter
   * \tparam constant The constant
   */
#ifdef __cpp_concepts
  template<indexible T, auto constant, vector_space_descriptor_collection Ds> requires value::number<decltype(constant)>
  constexpr constant_matrix auto
#else
  template<typename T, auto constant, typename Ds, std::enable_if_t<
    indexible<T> and value::number<decltype(constant)> and vector_space_descriptor_collection<Ds>, int> = 0>
  constexpr auto
#endif
  make_constant(Ds&& ds)
  {
    return make_constant<T, decltype(constant), constant>(std::forward<Ds>(ds));
  }


  /**
   * \overload
   * \brief Make a compile-time constant based on a particular library object and a scalar constant value known at compile time
   * \tparam T A matrix or tensor from a particular library.
   * \tparam C A \ref value::scalar for the new zero matrix. Must be constructible from {constant...}
   * \tparam constant A constant or set of coefficients in a vector space defining a constant
   * (e.g., real and imaginary parts of a complex number).
   * \param Ds A set of \ref vector_space_descriptor defining the dimensions of each index.
   */
#ifdef __cpp_concepts
  template<indexible T, value::scalar C, auto...constant, vector_space_descriptor...Ds> requires
    ((value::static_scalar<C> and sizeof...(constant) == 0) or requires { C {constant...}; })
  constexpr constant_matrix auto
#else
  template<typename T, typename C, auto...constant, typename...Ds, std::enable_if_t<
    indexible<T> and value::scalar<C> and (vector_space_descriptor<Ds> and ...) and
    ((value::static_scalar<C> and sizeof...(constant) == 0) or
      std::is_constructible<C, decltype(constant)...>::value), int> = 0>
  constexpr auto
#endif
  make_constant(Ds&&...ds)
  {
    return make_constant<T, C, constant...>(std::forward_as_tuple(std::forward<Ds>(ds)...));
  }


  /**
   * \overload
   * \brief Same as above, except that the scalar type is derived from the constant template parameter
   * \tparam constant The constant
   */
#ifdef __cpp_concepts
  template<indexible T, auto constant, vector_space_descriptor...Ds> requires value::number<decltype(constant)>
  constexpr constant_matrix auto
#else
  template<typename T, auto constant, typename...Ds, std::enable_if_t<
    indexible<T> and value::number<decltype(constant)> and (vector_space_descriptor<Ds> and ...), int> = 0>
  constexpr auto
#endif
  make_constant(Ds&&...ds)
  {
    return make_constant<T, decltype(constant), constant>(std::forward_as_tuple(std::forward<Ds>(ds)...));
  }


  /**
   * \overload
   * \brief Construct a constant object, where the shape of the new object is derived from t.
   */
#ifdef __cpp_concepts
  template<value::scalar C, auto...constant, indexible T> requires
    ((value::static_scalar<C> and sizeof...(constant) == 0) or requires { C {constant...}; })
  constexpr constant_matrix auto
#else
  template<typename C, auto...constant, typename T, std::enable_if_t<value::scalar<C> and indexible<T> and
    ((value::static_scalar<C> and sizeof...(constant) == 0) or std::is_constructible<C, decltype(constant)...>::value), int> = 0>
  constexpr auto
#endif
  make_constant(const T& t)
  {
    return make_constant<T, C, constant...>(all_vector_space_descriptors(t));
  }


/**
 * \overload
 * \brief Same as above, except that the scalar type is derived from the constant template parameter
 */
#ifdef __cpp_concepts
  template<auto constant, indexible T> requires value::number<decltype(constant)>
  constexpr constant_matrix auto
#else
  template<auto constant, typename T, std::enable_if_t<value::number<decltype(constant)> and indexible<T>, int> = 0>
  constexpr auto
#endif
  make_constant(const T& t)
  {
    return make_constant<decltype(constant), constant>(all_vector_space_descriptors(t));
  }


} // namespace OpenKalman

#endif //OPENKALMAN_MAKE_CONSTANT_HPP
