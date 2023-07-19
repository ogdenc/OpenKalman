/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_FIXEDSIZEADAPTER_HPP
#define OPENKALMAN_FIXEDSIZEADAPTER_HPP

namespace OpenKalman::internal
{
#ifdef __cpp_concepts
  template<indexible NestedMatrix, index_descriptor...IndexDescriptors>
    requires compatible_with_index_descriptors<NestedMatrix, IndexDescriptors...> and
      (sizeof...(IndexDescriptors) == max_indices_of_v<NestedMatrix>) and (not std::is_reference_v<NestedMatrix>)
#else
  template<typename NestedMatrix, typename...IndexDescriptors>
#endif
  struct FixedSizeAdapter : library_base<FixedSizeAdapter<NestedMatrix, IndexDescriptors...>, NestedMatrix>
  {

#ifndef __cpp_concepts
    static_assert(indexible<NestedMatrix>);
    static_assert((index_descriptor<IndexDescriptors> and ...));
    static_assert(compatible_with_index_descriptors<NestedMatrix, IndexDescriptors...>);
    static_assert(sizeof...(IndexDescriptors) == max_indices_of_v<NestedMatrix>);
    static_assert(not std::is_reference_v<NestedMatrix>);
#endif


    /**
     * \brief Construct by moving from compatible indexible object.
     */
#ifdef __cpp_concepts
    template<compatible_with_index_descriptors<IndexDescriptors...> Arg>
      requires std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<compatible_with_index_descriptors<Arg, IndexDescriptors...> and
      std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    explicit FixedSizeAdapter(Arg&& arg) noexcept : m_arg {std::forward<Arg>(arg)} {}


    /**
     * \brief Construction from an lvalue reference is not allowed.
     */
    template<typename Arg>
    FixedSizeAdapter(Arg& arg) = delete;


    /**
     * \brief Assign from another compatible indexible object.
     */
#ifdef __cpp_concepts
    template<compatible_with_index_descriptors<IndexDescriptors...> Arg> requires std::assignable_from<NestedMatrix&, Arg&&>
#else
    template<typename Arg, std::enable_if_t<compatible_with_index_descriptors<Arg, IndexDescriptors...> and std::is_assignable_v<NestedMatrix&, Arg&&>, int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      m_arg = to_native_matrix<NestedMatrix>(std::forward<Arg>(arg));
      return *this;
    }


    /**
     * \brief Get the nested matrix.
     */
    const NestedMatrix& nested_matrix() & noexcept { return m_arg; }

    /// \overload
    const NestedMatrix& nested_matrix() const & noexcept { return m_arg; }

    /// \overload
    NestedMatrix nested_matrix() && noexcept { return std::move(m_arg); }

    /// \overload
    NestedMatrix nested_matrix() const && noexcept { return std::move(m_arg); }


    /**
     * \brief Increment from another indexible object.
     */
#ifdef __cpp_concepts
    template<compatible_with_index_descriptors<IndexDescriptors...> Arg>
#else
    template<typename Arg, std::enable_if_t<compatible_with_index_descriptors<Arg, IndexDescriptors...>, int> = 0>
#endif
    auto& operator+=(Arg&& arg) noexcept
    {
      this->nested_matrix() += to_native_matrix<NestedMatrix>(std::forward<Arg>(arg));
      return *this;
    }


    /**
     * \brief Decrement from another indexible object.
     */
#ifdef __cpp_concepts
    template<compatible_with_index_descriptors<IndexDescriptors...> Arg>
#else
    template<typename Arg, std::enable_if_t<compatible_with_index_descriptors<Arg, IndexDescriptors...>, int> = 0>
#endif
    auto& operator-=(Arg&& arg) noexcept
    {
      this->nested_matrix() -= to_native_matrix<NestedMatrix>(std::forward<Arg>(arg));
      return *this;
    }

  private:

    NestedMatrix m_arg; //< The nested matrix.

  };


} // namespace OpenKalman::internal


// ------------------------- //
//        Interfaces         //
// ------------------------- //

namespace OpenKalman::interface
{
  template<typename NestedMatrix, typename IndexDescriptor, typename...IndexDescriptors>
  struct IndexTraits<internal::FixedSizeAdapter<NestedMatrix, IndexDescriptor, IndexDescriptors...>>
  {
    static constexpr std::size_t max_indices = 1 + sizeof...(IndexDescriptors);

    template<std::size_t N, typename Arg>
    static constexpr auto get_index_descriptor(const Arg& arg)
    {
      using ID = std::tuple_element_t<N, std::tuple<IndexDescriptor, IndexDescriptors...>>;
      if constexpr (fixed_index_descriptor<ID>) return ID {};
      else return OpenKalman::get_index_descriptor<N>(arg.nested_matrix());
    }
  };


  template<typename NestedMatrix, typename...IndexDescriptors>
  struct Elements<internal::FixedSizeAdapter<NestedMatrix, IndexDescriptors...>>
  {
    using scalar_type = scalar_type_of_t<NestedMatrix>;

#ifdef __cpp_lib_concepts
    template<typename Arg, typename...I> requires element_gettable<nested_matrix_of_t<Arg>, sizeof...(I)>
#else
    template<typename Arg, typename...I, std::enable_if_t<element_gettable<typename nested_matrix_of<Arg>::type, sizeof...(I)>, int> = 0>
#endif
    static constexpr decltype(auto) get(Arg&& arg, I...i)
    {
      return get_element(std::forward<Arg>(arg).nested_matrix(), i...);
    }
  };


  template<typename NestedMatrix, typename...IndexDescriptors>
  struct EquivalentDenseWritableMatrix<internal::FixedSizeAdapter<NestedMatrix, IndexDescriptors...>>
    : EquivalentDenseWritableMatrix<NestedMatrix>
  {
    template<typename Arg>
    static decltype(auto) convert(Arg&& arg)
    {
      return make_dense_writable_matrix_from(std::forward<Arg>(arg).nested_matrix());
    }

    template<typename Arg>
    static decltype(auto) to_native_matrix(Arg&& arg)
    {
      return OpenKalman::to_native_matrix<NestedMatrix>(std::forward<Arg>(arg).nested_matrix());
    }
  };


  template<typename NestedMatrix, typename...IndexDescriptors>
  struct Dependencies<internal::FixedSizeAdapter<NestedMatrix, IndexDescriptors...>>
  {
    static constexpr bool has_runtime_parameters = false;

    using type = std::tuple<NestedMatrix>;

    template<std::size_t i, typename Arg>
    static decltype(auto) get_nested_matrix(Arg&& arg)
    {
      static_assert(i == 0);
      return std::forward<Arg>(arg).nested_matrix();
    }

    template<typename Arg>
    static auto convert_to_self_contained(Arg&& arg)
    {
      return make_self_contained(std::forward<Arg>(arg).nested_matrix());
    }
  };


  template<typename NestedMatrix, typename...IndexDescriptors>
  struct SingleConstantMatrixTraits<internal::FixedSizeAdapter<NestedMatrix, IndexDescriptors...>>
    : SingleConstantMatrixTraits<NestedMatrix> {};


  template<typename NestedMatrix, typename...IndexDescriptors>
  struct SingleConstant<internal::FixedSizeAdapter<NestedMatrix, IndexDescriptors...>>
    : SingleConstant<NestedMatrix>
  {
    SingleConstant(const internal::FixedSizeAdapter<NestedMatrix, IndexDescriptors...>& arg)
      : SingleConstant<std::decay_t<NestedMatrix>> {arg.nested_matrix()} {};
  };


  template<typename NestedMatrix, typename...IndexDescriptors>
  struct SingleConstantDiagonalMatrixTraits<internal::FixedSizeAdapter<NestedMatrix, IndexDescriptors...>>
    : SingleConstantDiagonalMatrixTraits<NestedMatrix> {};


  template<typename NestedMatrix, typename...IndexDescriptors>
  struct TriangularTraits<internal::FixedSizeAdapter<NestedMatrix, IndexDescriptors...>>
    : TriangularTraits<NestedMatrix>
  {
    static constexpr bool is_triangular_adapter = false;

    template<Likelihood b>
    static constexpr bool is_diagonal_adapter = false;

  };


  template<typename NestedMatrix, typename...IndexDescriptors>
  struct HermitianTraits<internal::FixedSizeAdapter<NestedMatrix, IndexDescriptors...>>
  {
    static constexpr bool is_hermitian = hermitian_matrix<NestedMatrix, Likelihood::maybe>;

    template<HermitianAdapterType t, typename Arg>
    static constexpr auto make_hermitian_adapter(Arg&& arg)
    {
      return make_hermitian_matrix<t>(std::forward<Arg>(arg).nested_matrix());
    }
  };


  template<typename NestedMatrix, typename...IndexDescriptors>
  struct Subsets<internal::FixedSizeAdapter<NestedMatrix, IndexDescriptors...>>
  {
    template<typename Arg, typename...Begin, typename...Size>
    static decltype(auto) get_block(Arg&& arg, std::tuple<Begin...> begin, std::tuple<Size...> size)
    {
      return OpenKalman::get_block(std::forward<Arg>(arg).nested_matrix(), begin, size);
    };

    template<typename Arg, typename Block, typename...Begin>
    static Arg& set_block(Arg& arg, Block&& block, Begin...begin)
    {
      return OpenKalman::set_block(std::forward<Arg>(arg).nested_matrix(), std::forward<Block>(block), begin...);
    };

    template<TriangleType t, typename A, typename B>
    static decltype(auto) set_triangle(A&& a, B&& b)
    {
      return OpenKalman::internal::set_triangle<t>(std::forward<A>(a).nested_matrix(), std::forward<B>(b));
    }
  };


  template<typename NestedMatrix, typename...IndexDescriptors>
  struct Conversions<internal::FixedSizeAdapter<NestedMatrix, IndexDescriptors...>>
  {
    template<typename Arg>
    static constexpr decltype(auto) to_diagonal(Arg&& arg)
    {
      return OpenKalman::to_diagonal(std::forward<Arg>(arg).nested_matrix());
    }

    template<typename Arg>
    static constexpr decltype(auto) diagonal_of(Arg&& arg)
    {
      return OpenKalman::diagonal_of(std::forward<Arg>(arg).nested_matrix());
    }
  };


  template<typename NestedMatrix, typename...IndexDescriptors>
  struct ArrayOperations<internal::FixedSizeAdapter<NestedMatrix, IndexDescriptors...>>
  {
    template<typename...Ds, typename Operation, typename...Args>
    static auto n_ary_operation(const std::tuple<Ds...>& d_tup, Operation&& op)
    {
      return ArrayOperations<NestedMatrix>::n_ary_operation(d_tup, std::forward<Operation>(op));
    }

    template<typename...Ds, typename Operation, typename Arg, typename...Args>
    static auto n_ary_operation(const std::tuple<Ds...>& d_tup, Operation&& op, Arg&& arg, Args&&...args)
    {
      return ArrayOperations<NestedMatrix>::n_ary_operation(d_tup, std::forward<Operation>(op), std::forward<Arg>(arg).nested_matrix(), std::forward<Args>(args)...);
    }

    template<typename...Ds, typename Operation, typename Arg, typename...Args>
    static auto n_ary_operation_with_indices(const std::tuple<Ds...>& d_tup, Operation&& op)
    {
      return ArrayOperations<NestedMatrix>::n_ary_operation_with_indices(d_tup, std::forward<Operation>(op));
    }

    template<typename...Ds, typename Operation, typename Arg, typename...Args>
    static auto n_ary_operation_with_indices(const std::tuple<Ds...>& d_tup, Operation&& op, Arg&& arg, Args&&...args)
    {
      return ArrayOperations<NestedMatrix>::n_ary_operation_with_indices(d_tup, std::forward<Operation>(op), std::forward<Arg>(arg).nested_matrix(), std::forward<Args>(args)...);
    }

    template<std::size_t...indices, typename BinaryFunction, typename Arg>
    static constexpr decltype(auto) reduce(BinaryFunction&& b, Arg&& arg)
    {
      return ArrayOperations<NestedMatrix>::template reduce<indices...>(std::forward<BinaryFunction>(b), std::forward<Arg>(arg).nested_matrix());
    }
  };


  template<typename NestedMatrix, typename...IndexDescriptors>
  struct ModularTransformationTraits<internal::FixedSizeAdapter<NestedMatrix, IndexDescriptors...>>
  {
    template<typename Arg, typename C>
    constexpr decltype(auto) to_euclidean(Arg&& arg, const C& c)
    {
      return to_euclidean(std::forward<Arg>(arg).nested_matrix(), c);
    }

    template<typename Arg, typename C>
    constexpr decltype(auto) from_euclidean(Arg&& arg, const C& c)
    {
      return from_euclidean(std::forward<Arg>(arg).nested_matrix(), c);
    }

    template<typename Arg, typename C>
    constexpr decltype(auto) wrap_angles(Arg&& arg, const C& c)
    {
      return wrap_angles(std::forward<Arg>(arg).nested_matrix(), c);
    }
  };


  template<typename NestedMatrix, typename...IndexDescriptors>
  struct LinearAlgebra<internal::FixedSizeAdapter<NestedMatrix, IndexDescriptors...>>
  {
    template<typename Arg>
    static constexpr auto conjugate(Arg&& arg) { return OpenKalman::conjugate(std::forward<Arg>(arg).nested_matrix()); }

    template<typename Arg>
    static constexpr auto transpose(Arg&& arg) { return OpenKalman::transpose(std::forward<Arg>(arg).nested_matrix()); }

    template<typename Arg>
    static constexpr auto adjoint(Arg&& arg) { return OpenKalman::adjoint(std::forward<Arg>(arg).nested_matrix()); }

    template<typename Arg>
    static constexpr auto determinant(Arg&& arg) { return OpenKalman::determinant(std::forward<Arg>(arg).nested_matrix()); }

    template<typename A, typename B>
    static constexpr auto sum(A&& a, B&& b) { return OpenKalman::sum(std::forward<A>(a).nested_matrix(), std::forward<B>(b)); }

    template<typename A, typename B>
    static constexpr auto contract(A&& a, B&& b) { return OpenKalman::contract(std::forward<A>(a).nested_matrix(), std::forward<B>(b)); }

    template<typename A, typename B>
    static constexpr auto contract_in_place(A&& a, B&& b) { return OpenKalman::contract_in_place(std::forward<A>(a).nested_matrix(), std::forward<B>(b)); }

    template<TriangleType t, typename Arg>
    static constexpr auto cholesky_factor(Arg&& arg) { return OpenKalman::Cholesky_factor<t>(std::forward<Arg>(arg).nested_matrix()); }

    template<typename A, typename U, typename Alpha>
    static constexpr auto rank_update_self_adjoint(A&& a, U&& u, Alpha alpha) { return OpenKalman::rank_update_self_adjoint(std::forward<A>(a).nested_matrix(), std::forward<U>(u), alpha); }

    template<TriangleType t, typename A, typename U, typename Alpha>
    static constexpr auto rank_update_triangular(A&& a, U&& u, Alpha alpha) { return OpenKalman::rank_update_triangular<t>(std::forward<A>(a).nested_matrix(), std::forward<U>(u), alpha); }

    template<bool must_be_unique = false, bool must_be_exact = false, typename A, typename B>
    static constexpr auto solve(A&& a, B&& b) { return OpenKalman::solve<must_be_unique, must_be_exact>(std::forward<A>(a).nested_matrix(), std::forward<B>(b)); }

    template<typename Arg>
    static constexpr auto LQ_decomposition(Arg&& arg) { return OpenKalman::LQ_decomposition(std::forward<Arg>(arg).nested_matrix()); }

    template<typename Arg>
    static constexpr auto QR_decomposition(Arg&& arg) { return OpenKalman::QR_decomposition(std::forward<Arg>(arg).nested_matrix()); }
  };

} // namespace interface



#endif //OPENKALMAN_FIXEDSIZEADAPTER_HPP
