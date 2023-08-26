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
 * \brief Forward declaration of IndexibleObjectTraits, which must be defined for all objects used in OpenKalman.
 */

#ifndef OPENKALMAN_INDEXIBLEOBJECTTRAITS_HPP
#define OPENKALMAN_INDEXIBLEOBJECTTRAITS_HPP

#include <type_traits>
#include <tuple>


namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief An interface to traits of a particular object (matrix, array, tensor, etc.) within a library.
   * \tparam T The matrix, array, expression, or tensor.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct IndexibleObjectTraits
  {
    /**
     * \brief The maximum number of indices by which the elements of T are accessible.
     * \details T may optionally be accessible by fewer indices.
     */
    static constexpr std::size_t max_indices = 0;


    /**
     * \brief Get an \ref index_descriptor for index N of an argument.
     * \tparam N The index
     * \param arg An indexible object (tensor, matrix, vector, etc.)
     * \return an \ref index_descriptor (either fixed or dynamic)
     */
#ifdef __cpp_concepts
    template<std::size_t N, std::convertible_to<const std::remove_reference_t<T>&> Arg>
    static constexpr index_descriptor auto get_index_descriptor(const Arg& arg)
#else
    template<std::size_t N, typename Arg, std::enable_if_t<std::is_convertible_v<Arg,
      const std::add_lvalue_reference_t<std::decay_t<T>>>, int> = 0>
    static constexpr auto get_index_descriptor(const Arg& arg)
#endif
    {
      return Dimensions<0>{};
    }


    /**
     * \brief Whether T is one-by-one (optional).
     * \details This can be useful because some dynamic matrix types erase the shape info about their nested matrix,
     * such that it does not preserve the knowledge that the object cannot be one-by-one.
     */
    template<Likelihood b>
    static constexpr bool is_one_by_one = false;


    /**
     * \brief Whether all dimensions of T are the same and type-equivalent (optional).
     * \details This can be useful because some dynamic matrix types erase the shape info about their nested matrix,
     * such that it does not preserve the knowledge that the object cannot be square.
     */
    template<Likelihood b>
    static constexpr bool is_square = false;


    /**
     * \brief Indicates whether type T stores any internal runtime parameters.
     * \details An example of an internal runtime parameter might be indices for start locations, or sizes, for an
     * expression representing a block or sub-matrix within a matrix. If unknown, the value of <code>true</code> is
     * the safest and will prevent unintended dangling references.
     * \note If this is not defined, T will be treated as if it is defined and true. This parameter can ignore whether
     * any nested matrices, themselves, have internal runtime parameters.
     */
    static constexpr bool has_runtime_parameters = true;


    /**
     * \brief Gets the i-th dependency of T.
     * /detail There is no need to check the bounds of <code>i</code>, but they should be treated as following this
     * constraint:
     * /code
     *   requires (i < std::tuple_size_v<type>) and std::same_as<std::decay_t<Arg>, std::decay_t<T>>
     * /endcode
     * \note Defining this function is optional. Also, there is no need for the example constraints on i or Arg,
     * as OpenKalman::nested_matrix already enforces these constraints.
     * \tparam i Index of the dependency (0 for the 1st dependency, 1 for the 2nd, etc.).
     * \tparam Arg An object of type T
     * \return The i-th dependency of T
     * \sa OpenKalman::nested_matrix
     */
#ifdef __cpp_concepts
    template<std::size_t i, std::convertible_to<const std::remove_reference_t<T>&> Arg>
#else
    template<std::size_t i, typename Arg, std::enable_if_t<
      std::is_convertible_v<Arg, const std::remove_reference_t<T>&>, int> = 0>
#endif
    static decltype(auto) get_nested_matrix(Arg&& arg) = delete;


    /**
     * \brief Converts an object of type T into an equivalent, self-contained object (i.e., no external dependencies).
     * \detail The resulting type must be equivalent to T, including in shape and scalar type. But it must be
     * self-contained, so that it has external dependencies accessible only by lvalue references. The result must be
     * guaranteed to be returnable from a function without causing a dangling reference. If possible, this should
     * preserve the traits of T, such as whether it is a \ref triangular_matrix, \ref diagonal_matrix, or
     * \note Defining this function is optional. If not defined, the default behavior is to convert to the equivalent,
     * dense, writable matrix. Also, there is no need for the example constraint on Arg, as
     * OpenKalman::make_self_contained already enforces this constraint.
     * \ref zero_matrix.
     * \tparam Arg An object of type T
     * \return An equivalent self-contained version of T
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const std::remove_reference_t<T>&> Arg>
#else
    template<typename Arg, std::enable_if_t<std::is_convertible_v<Arg, const std::remove_reference_t<T>&>, int> = 0>
#endif
    static decltype(auto) convert_to_self_contained(Arg&& arg) = delete;


    /**
     * \typedef type
     * \brief A tuple with elements corresponding to each dependent object.
     * \details If the object is linked within T by an lvalue reference, the element should be an lvalue reference.
     * Examples:
     * \code
     *   using type = std::tuple<>; //< T has no dependencies
     *   using type = std::tuple<Arg1, Arg2&>; //< T stores Arg1 and a reference to Arg2
     * \endcode
     * \note If this is not defined, T will be considered non-self-contained.
     */


    /**
     * \brief If the instance of T has a constant value across all elements, return that constant.
     * \details The return type must be convertible to scalar_type_of_t<T>. If T's elements are a constant known
     * only at runtime, this should be a non-static function.
     */
    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg) = delete;


    /**
     * \brief If the instance of T has a constant diagonal value, return that constant.
     * \details The return type must be convertible to scalar_type_of_t<T>. If T's elements are a constant known
     * only at runtime, this should be a non-static function.
     */
    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg) = delete;


    /**
     * \brief Whether T is triangular or diagonal, having a triangle type of t.
     * \details This trait should propagate from any nested matrices or matrices involved in any expression arguments.
     * \tparam t The \ref TriangleType
     * \tparam b The \ref Likelihood. If <code>b == Likelihood::definitely</code>, then T's triangle type is known at compile time.
     * If <code>b == Likelihood::maybe</code>, then T's triangle type is determined at runtime (for example, T might be
     * triangular if and only iff it is a square matrix, but it is unknown whether T is square).
     */
    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = false;

    /**
     * \brief Whether T is a triangular adapter (defaults to false, if omitted).
     * \details This is not a guarantee that the matrix is triangular, because it could be dynamically non-square.
     */
    static constexpr bool is_triangular_adapter = false;

    /**
     * \brief Whether T is a \ref diagonal_adapter (defaults to false, if omitted).
     * \details The likelihood b is available if it is not known whether the nested matrix is a column vector
     */
    template<Likelihood b>
    static constexpr bool is_diagonal_adapter = false;

    /**
     * \brief Create a \ref triangular_matrix from a square matrix.
     * \details This is used by the function OpenKalman::make_triangular_matrix. This can be left undefined if
     * - Arg is already triangular and of a TriangleType compatible with t, or
     * - the intended result is for Arg to be wrapped in an \ref Eigen::TriangularMatrix (which will happen automatically).
     * \tparam t The intended \ref TriangleType of the result.
     * \tparam Arg A square matrix to be made triangular.
     */
#ifdef __cpp_concepts
    template<TriangleType t, std::convertible_to<const std::remove_reference_t<T>&> Arg>
#else
    template<TriangleType t, typename Arg, std::enable_if_t<std::is_convertible_v<Arg, const std::remove_reference_t<T>&>, int> = 0>
#endif
    static constexpr auto make_triangular_matrix(Arg&& arg) = delete;


    /**
     * \brief Whether T is hermitian.
     */
    static constexpr bool is_hermitian = false;


    /**
     * \brief The hermitian-adapter storage type of T, if any (optional).
     * \details This is not a guarantee that the matrix is hermitian, because it could be dynamically non-square.
     * If T is not a hermitian adapter, this should be omitted. Permissible values are HermitianAdapterType::upper and
     * HermitianAdapterType::lower.
     */
     //static constexpr HermitianAdapterType adapter_type = HermitianAdapterType::lower;

     /**
      * \brief Make a hermitian adapter.
      */
     template<HermitianAdapterType t, typename Arg>
     static constexpr auto make_hermitian_adapter(Arg&& arg) = delete;


    /**
     * \typedef scalar_type
     * \brief The scalar type of T (e.g., double, int).
     * \details Example:
     * \code
     * using scalar_type = double;
     * \endcode
     */


    /**
     * \brief Get the element at indices (i...) of the object. This should preferably return a non-const lvalue reference, if possible.
     * \details This function, or the library, is responsible for any bounds checking.
     * \returns an element or reference to an element
     *
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const std::decay_t<T>&> Arg, std::convertible_to<const std::size_t>...I>
#else
    template<typename Arg, typename...I, std::enable_if_t<
      std::is_convertible_v<Arg, const std::decay_t<T>&> and (std::is_convertible_v<I, const std::size_t> and ...), int> = 0>
#endif
    static constexpr decltype(auto) get(Arg&& arg, I...i) = delete;


    /**
     * \brief Set element at indices (i...) of the object to s.
     * \details This function, or the library, is responsible for any bounds checking.
     *
     */
#ifdef __cpp_concepts
    template<std::convertible_to<std::decay_t<T>> Arg, std::convertible_to<const std::size_t>...I>
      requires (not std::is_const_v<std::remove_reference_t<Arg>>)
#else
    template<typename Arg, typename...I, std::enable_if_t<
      std::is_convertible_v<Arg, std::decay_t<T>> and (std::is_convertible_v<I, const std::size_t> and ...) and
      not std::is_const_v<std::remove_reference_t<Arg>>, int> = 0>
#endif
    static void set(Arg& arg, const typename IndexibleObjectTraits<std::decay_t<Arg>>::scalar_type& s, I...i) = delete;


    /**
     * \brief Whether T is a writable, self-contained matrix or array.
     */
    static constexpr bool is_writable = false;


    /**
     * \brief If the argument has direct access to the underlying array data, return a pointer to that data.
     */
    template<typename Arg>
    static constexpr auto*
    data(Arg& arg) = delete;


    /**
     * \brief The layout of T.
     */
    static constexpr Layout layout = Layout::none;


    /**
     * \brief If layout is Layout::stride, this returns a tuple of strides, one for each dimension.
     * \details This is only necessary or meaningful if layout == Layout::stride.
     * The tuple elements may be integral constants if the values are known at compile time. Example:
     * <code>return std::tuple {16, 4, std::integral_constant<std::size_t, 1>{}></code>
     */
    template<typename Arg>
    static constexpr auto
    strides(Arg&& arg) = delete;

  };

} // namespace OpenKalman::interface


#endif //OPENKALMAN_INDEXIBLEOBJECTTRAITS_HPP
